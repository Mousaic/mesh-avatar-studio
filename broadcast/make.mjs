// Batch broadcast videos: each script .txt -> Edge TTS speech -> avatar rendered frame by frame -> MP4.
//
//   node broadcast/make.mjs <project> <script.txt | folder> [--engine edge|doubao] [--voice ...] [--rate +0%] [--no-tone]
//        [--bg #0b1f3a | path/to/background.png] [--size 1080x1920] [--fps 30]
//
// Videos go to <project>/videos/<script name>.mp4; speech and timelines to <project>/work/broadcast/.
// Only the script text leaves the machine (to the speech service); the avatar is rendered locally.
import { readFile, readdir, mkdir, realpath, stat, writeFile } from 'node:fs/promises';
import { resolve, relative, dirname, basename, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'vite';
import { chromium } from '@playwright/test';
import { validateRig } from '../packages/runtime/src/rig/validate.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const here = dirname(fileURLToPath(import.meta.url));
const PY = ['run', '-q', '--with', 'edge-tts', '--with', 'websockets', '--with', 'pypinyin', '--with', 'imageio-ffmpeg', '--with', 'numpy', 'python', '-I'];

function parseArgs(argv) {
  const options = { engine: 'edge', rate: '+0%', bg: '#0b1f3a', size: '1080x1920', fps: '30', lip: 'news' };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--no-tone') options.noTone = true;
    else if (argv[i] === '--reuse-speech') options.reuseSpeech = true;
    else if (argv[i].startsWith('--')) options[argv[i].slice(2)] = argv[++i];
    else positional.push(argv[i]);
  }
  if (positional.length !== 2) throw new Error('Usage: node broadcast/make.mjs <project> <script.txt | folder> [--voice ...] [--rate +0%] [--bg color|image] [--size 1080x1920] [--fps 30]');
  const [width, height] = options.size.split('x').map(Number);
  return { project: positional[0], input: positional[1], ...options, width, height, fps: Number(options.fps), lipSync: lipStyle(options.lip) };
}

// Mouth movement styles of the studio's Lip sync tab (src/editor/LipSyncSettings.tsx), or "g,o,c" numbers.
// "news" (default) separates only clear syllable dips and closes at a moderate speed: about a fifth
// of the full open-close cycles of "default", while the mouth still shuts in pauses.
const LIP_STYLES = {
  news: { granularity: 0.15, openness: 0.5, closeSpeed: 0.35 },
  calm: { granularity: 0, openness: 0.5, closeSpeed: 0 },
  default: { granularity: 0.5, openness: 0.5, closeSpeed: 0.5 },
  lively: { granularity: 1, openness: 0.5, closeSpeed: 1 },
  subtle: { granularity: 0.5, openness: 0, closeSpeed: 0.5 },
  big: { granularity: 0.5, openness: 1, closeSpeed: 0.5 },
};
function lipStyle(value) {
  if (LIP_STYLES[value]) return LIP_STYLES[value];
  const [granularity, openness, closeSpeed] = value.split(',').map(Number);
  if (![granularity, openness, closeSpeed].every(n => n >= 0 && n <= 1)) throw new Error(`--lip: one of ${Object.keys(LIP_STYLES).join(', ')}, or three numbers 0..1 like 0.2,0.5,0.1`);
  return { granularity, openness, closeSpeed };
}

async function scriptsOf(input) {
  const path = resolve(input);
  if ((await stat(path)).isFile()) return [path];
  return (await readdir(path)).filter(name => extname(name) === '.txt').sort().map(name => resolve(path, name));
}

/** The avatar's built files as data URLs, as tools/render-poses.mjs loads them. */
async function avatarAssets(project) {
  const built = await realpath(resolve(project, 'built'));
  const assets = {};
  const add = async name => {
    const path = await realpath(resolve(built, name));
    if (relative(built, path).startsWith('..')) throw new Error(`asset escapes built folder: ${name}`);
    assets[name] = `data:${name.endsWith('.png') ? 'image/png' : 'application/json'};base64,${(await readFile(path)).toString('base64')}`;
  };
  await add('layers.json');
  const layers = JSON.parse(await readFile(resolve(built, 'layers.json'), 'utf8'));
  await Promise.all(['base', 'hairmask', ...Object.keys(layers.layers)].map(name => add(`${name}.png`)));
  try {
    await add('sprites/sprites.json');
    const sprites = JSON.parse(await readFile(resolve(built, 'sprites/sprites.json'), 'utf8'));
    await Promise.all(Object.keys(sprites.layers).map(name => add(`sprites/${name}.png`)));
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  return assets;
}

async function background(bg) {
  if (/^#[0-9a-f]{3,8}$/i.test(bg)) return { color: bg };
  const data = await readFile(resolve(bg));
  return { image: `data:image/${extname(bg).slice(1).replace('jpg', 'jpeg')};base64,${data.toString('base64')}` };
}

// Runs inside the page: builds the scene once, then renders frames on request.
async function setupScene({ rig, assets, width, height, bg }) {
  const { createMeshAvatar, PARAMS } = await import('/packages/runtime/src/index.ts');
  // Frame the drawn part of the avatar: its cut-off bottom edge goes just below the frame and
  // the top of the head sits under the title band, without cropping it wider than the frame.
  const base = new Image(); base.src = assets['base.png']; await base.decode();
  const probe = document.createElement('canvas'); probe.width = base.width; probe.height = base.height;
  const pctx = probe.getContext('2d', { willReadFrequently: true }); pctx.drawImage(base, 0, 0);
  const alpha = pctx.getImageData(0, 0, base.width, base.height).data;
  let x0 = base.width, x1 = 0, y0 = base.height, y1 = 0;
  for (let y = 0; y < base.height; y++) for (let x = 0; x < base.width; x++) {
    if (alpha[(y * base.width + x) * 4 + 3] > 8) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  const IMG = { w: rig.image.width, h: rig.image.height }, kx = IMG.w / base.width, ky = IMG.h / base.height;
  const headTop = height * 0.12;
  const scale = Math.min((height - headTop) / ((y1 - y0) * ky), width * 1.08 / ((x1 - x0) * kx));
  const avatarBox = { w: Math.round(IMG.w * scale), h: Math.round(IMG.h * scale) };
  avatarBox.x = Math.round(width / 2 - (x0 + x1) / 2 * kx * scale);
  avatarBox.y = Math.round(height + 6 - y1 * ky * scale);
  const avatarCanvas = document.getElementById('avatar');
  avatarCanvas.style.cssText = `position:absolute;left:0;top:0;width:${avatarBox.w}px;height:${avatarBox.h}px`;
  const out = document.createElement('canvas'); out.width = width; out.height = height;
  const ctx = out.getContext('2d');
  let bgImage = null;
  if (bg.image) { bgImage = new Image(); bgImage.src = bg.image; await bgImage.decode(); }
  const defaults = Object.fromEntries(PARAMS.map(p => [p.id, p.def]));
  window.scene = { createMeshAvatar, defaults, rig, assets, avatarCanvas, avatarBox, out, ctx, bg, bgImage, width, height };
}

async function startClip(timeline, lipSync) {
  const s = window.scene;
  s.avatar?.destroy();
  s.avatar = await s.createMeshAvatar(s.avatarCanvas, { rig: s.rig, assets: s.assets, manual: true, padTop: 0, padSide: 0, preserveMouthForm: true, lipSync });
  s.avatar.setAutoIdle(true); s.avatar.setAutoMotion(true);
  s.avatar.advance(1, timeline.fps);          // settle physics before the first frame
  s.timeline = timeline; s.frame = 0;
  s.marks = [...(timeline.expressions ?? [])].sort((a, b) => a.time - b.time); s.wink = null;
  s.trace = [];
  const spoken = timeline.frames.map((f, i) => f[1] !== null ? i : -1).filter(i => i >= 0);
  s.speech = [spoken[0] ?? 0, spoken[spoken.length - 1] ?? 0];
}

function wrap(ctx, text, maxWidth) {
  const lines = []; let line = '';
  for (const ch of text) {
    if (ctx.measureText(line + ch).width > maxWidth && line) { lines.push(line); line = ch; } else line += ch;
  }
  if (line) lines.push(line);
  return lines;
}

function renderFrames(count) {
  const s = window.scene, t = s.timeline, { ctx, width: W, height: H } = s, images = [];
  for (let n = 0; n < count && s.frame < t.frames.length; n++, s.frame++) {
    const [level, form] = t.frames[s.frame], time = s.frame / t.fps;
    const speaking = s.frame >= s.speech[0] - 3 && s.frame <= s.speech[1] + 3;
    s.avatar.setSpeaking(speaking);
    s.avatar.setVoiceLevel(speaking ? level : 0);
    s.avatar.setParameters(form === null ? {} : { mouthForm: form });
    // Expression marks: emotions change the talking face and play a gesture; the rest are overlays.
    // Angry plays no gesture: its head shake closes the eyes into a playful ^^.
    while (s.marks.length && s.marks[0].time <= time) {
      const { name } = s.marks.shift();
      if (['happy', 'surprised', 'sad', 'angry', 'relaxed'].includes(name)) { s.avatar.setEmotion(name, { playMotion: name !== 'angry' }); s.avatar.setExpression('neutral'); }
      else if (name === 'neutral') { s.avatar.setEmotion('neutral', { playMotion: false }); s.avatar.setExpression('neutral'); }
      else { s.avatar.setExpression(name); s.wink = name === 'wink' ? time + 0.7 : null; }
    }
    if (s.wink !== null && time >= s.wink) { s.avatar.setExpression('neutral'); s.wink = null; }
    s.avatar.advance(1 / t.fps, t.fps);
    const P = s.avatar.getParameters();
    s.trace.push([+P.mouthOpen.toFixed(3), +P.mouthForm.toFixed(3)]);

    // background
    if (s.bgImage) {
      const k = Math.max(W / s.bgImage.width, H / s.bgImage.height);
      ctx.drawImage(s.bgImage, (W - s.bgImage.width * k) / 2, (H - s.bgImage.height * k) / 2, s.bgImage.width * k, s.bgImage.height * k);
    } else {
      const g = ctx.createLinearGradient(0, 0, 0, H);
      g.addColorStop(0, s.bg.color); g.addColorStop(1, '#16457a');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
    const box = s.avatarBox;
    ctx.drawImage(s.avatarCanvas, box.x, box.y, box.w, box.h);

    // title band
    if (t.title) {
      ctx.fillStyle = 'rgba(190, 20, 35, 0.92)'; ctx.fillRect(0, H * 0.035, W, H * 0.06);
      ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.round(H * 0.03)}px "Microsoft YaHei", "PingFang SC", sans-serif`;
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillText(t.title, W * 0.05, H * 0.065);
    }

    // subtitle
    const sub = t.subtitles.find(x => time >= x.start && time < x.end);
    if (sub) {
      const size = Math.round(W * 0.058);
      ctx.font = `bold ${size}px "Microsoft YaHei", "PingFang SC", sans-serif`;
      const lines = wrap(ctx, sub.text, W * 0.86), lh = size * 1.35, boxH = lines.length * lh + size * 0.8;
      const y0 = H * 0.86 - boxH / 2;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
      ctx.beginPath(); ctx.roundRect(W * 0.04, y0, W * 0.92, boxH, size * 0.4); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      lines.forEach((line, i) => ctx.fillText(line, W / 2, y0 + size * 0.4 + lh * (i + 0.5)));
    }
    images.push(s.out.toDataURL('image/jpeg', 0.92).split(',')[1]);
  }
  return images;
}

function ffmpegPath() {
  return execFileSync('uv', ['run', '-q', '--with', 'imageio-ffmpeg', 'python', '-I', '-c', 'import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())'], { encoding: 'utf8' }).trim();
}

function run(command, args, options = {}) {
  return new Promise((ok, fail) => {
    const child = spawn(command, args, { stdio: 'inherit', ...options });
    child.on('error', fail);
    child.on('exit', code => code === 0 ? ok() : fail(new Error(`${basename(command)} exited with ${code}`)));
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const project = await realpath(resolve(args.project));
  if (relative(root, project).startsWith('..')) throw new Error('project must stay inside the repository');
  const rig = JSON.parse(await readFile(resolve(project, 'rig.json'), 'utf8'));
  const errors = validateRig(rig);
  if (errors.length) throw new Error(errors.join('\n'));
  const scripts = await scriptsOf(args.input);
  if (!scripts.length) throw new Error(`no .txt scripts in ${args.input}`);
  const ffmpeg = ffmpegPath();
  const videos = resolve(project, 'videos');
  await mkdir(videos, { recursive: true });

  const server = await createServer({ root, configFile: false, publicDir: false, appType: 'custom', cacheDir: 'node_modules/.vite-broadcast', optimizeDeps: { noDiscovery: true, entries: [] },
    plugins: [{ name: 'broadcast-page', configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url !== '/__broadcast__') return next();
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><html><body style="margin:0;overflow:hidden"><canvas id="avatar"></canvas></body></html>');
      });
    } }],
    server: { host: '127.0.0.1', port: 0, strictPort: false, hmr: false, watch: null, preTransformRequests: false }, logLevel: 'error' });
  let browser;
  try {
    await server.listen();
    const origin = `http://127.0.0.1:${server.httpServer.address().port}`;
    browser = await chromium.launch({ args: ['--use-angle=d3d11', '--enable-gpu'] });
    const page = await browser.newPage({ viewport: { width: args.width, height: args.height }, deviceScaleFactor: 1 });
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.route('**/*', route => new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    await page.goto(`${origin}/__broadcast__`);
    await page.evaluate(`window.wrap = ${wrap}; window.startClip = ${startClip}; window.renderFrames = ${renderFrames};`);
    await page.evaluate(setupScene, { rig, assets: await avatarAssets(project), width: args.width, height: args.height, bg: await background(args.bg) });

    for (const script of scripts) {
      const name = basename(script, '.txt');
      const work = resolve(project, 'work', 'broadcast', name);
      console.log(`\n[${name}] speech (${args.engine}${args.voice ? `, ${args.voice}` : ''}${args.reuseSpeech ? ', reusing earlier audio when present' : ''})`);
      await run('uv', [...PY, resolve(here, 'tts.py'), script, work, '--engine', args.engine, ...(args.voice ? ['--voice', args.voice] : []),
        `--rate=${args.rate}`, '--fps', String(args.fps), ...(args.noTone ? ['--no-tone'] : []), ...(args.reuseSpeech ? ['--reuse'] : [])]);
      const timeline = JSON.parse(await readFile(resolve(work, 'timeline.json'), 'utf8'));
      await page.evaluate(([timeline, lipSync]) => window.startClip(timeline, lipSync), [timeline, args.lipSync]);

      const output = resolve(videos, `${name}.mp4`);
      const encoder = spawn(ffmpeg, ['-v', 'error', '-y', '-f', 'image2pipe', '-c:v', 'mjpeg', '-framerate', String(args.fps), '-i', 'pipe:0',
        '-i', resolve(work, 'speech.wav'), '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', output], { stdio: ['pipe', 'inherit', 'inherit'] });
      const done = new Promise((ok, fail) => { encoder.on('error', fail); encoder.on('exit', code => code === 0 ? ok() : fail(new Error(`ffmpeg exited with ${code}`))); });
      const started = Date.now();
      let written = 0;
      while (written < timeline.frames.length) {
        const images = await page.evaluate(n => window.renderFrames(n), 15);
        if (pageErrors.length) throw new Error(pageErrors.join('\n'));
        for (const image of images) {
          if (!encoder.stdin.write(Buffer.from(image, 'base64'))) await new Promise(ok => encoder.stdin.once('drain', ok));
        }
        written += images.length;
        process.stdout.write(`\r[${name}] frames ${written}/${timeline.frames.length}`);
      }
      encoder.stdin.end();
      await done;
      // mouth open / form per frame, for checking how busy the mouth is
      await writeFile(resolve(work, 'mouth.json'), JSON.stringify(await page.evaluate(() => window.scene.trace)));
      console.log(`\n[${name}] ${timeline.duration.toFixed(1)} s video in ${((Date.now() - started) / 1000).toFixed(0)} s -> ${relative(root, output)}`);
    }
  } finally {
    await browser?.close();
    await server.close();
  }
}

main().catch(error => { console.error(error.message ?? error); process.exitCode = 1; });
