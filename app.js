'use strict';
const videos = [...document.querySelectorAll('video')];
const states = new WeakMap();

function suspend(video) {
  const state = states.get(video);
  if (!video.paused) {
    state.automaticPauses += 1;
    video.pause();
  }
}

function synchronize(video) {
  const state = states.get(video);
  if (document.hidden) {
    suspend(video);
    return;
  }
  if (!state.wantsPlayback || !video.paused || state.starting) return;
  state.starting = true;
  video.play().then(() => {
    state.retry.hidden = true;
    if (document.hidden) suspend(video);
  }).catch(error => {
    // Autoplay can be denied on mobile or in embedded browsers. Offer a real
    // user-gesture play action rather than leaving a silent frozen poster.
    if (error.name !== 'AbortError') state.retry.hidden = false;
  }).finally(() => { state.starting = false; });
}

for (const video of videos) {
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
  video.autoplay = true;
  video.loop = true;
  video.preload = 'auto';
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'video-play-fallback';
  retry.textContent = 'Play video';
  retry.setAttribute('aria-label', `Play ${video.getAttribute('aria-label') || 'video'}`);
  retry.hidden = true;
  video.after(retry);
  const state = { wantsPlayback: true, automaticPauses: 0, starting: false, retry };
  states.set(video, state);
  retry.addEventListener('click', () => {
    state.wantsPlayback = true;
    // Call play directly inside the gesture for Safari / embedded browsers.
    video.play().then(() => { retry.hidden = true; }).catch(() => {
      retry.textContent = 'Open video';
      retry.onclick = () => window.open(video.currentSrc || video.querySelector('source').src, '_blank', 'noopener');
    });
  });
  video.addEventListener('play', () => {
    state.wantsPlayback = true;
    retry.hidden = true;
  });
  video.addEventListener('pause', () => {
    if (state.automaticPauses) state.automaticPauses -= 1;
    else state.wantsPlayback = false; // Preserve a visitor's manual pause.
  });
}

const synchronizeAll = () => videos.forEach(synchronize);
document.addEventListener('visibilitychange', synchronizeAll);
document.addEventListener('fullscreenchange', synchronizeAll);
window.addEventListener('pageshow', synchronizeAll);
window.addEventListener('focus', synchronizeAll);
window.addEventListener('pagehide', () => videos.forEach(suspend));

// Start every clip immediately, including clips below the fold.
synchronizeAll();
// A browser that blocks autoplay can unlock every clip on the first gesture.
for (const event of ['pointerdown', 'keydown']) {
  document.addEventListener(event, e => {
    if (!e.target.closest('video,button,a')) synchronizeAll();
  });
}
