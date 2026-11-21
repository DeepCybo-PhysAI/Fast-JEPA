'use strict';
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
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
  const fullscreen = document.fullscreenElement === video;
  if (document.hidden || (!state.visible && !fullscreen)) {
    suspend(video);
    return;
  }
  if (!state.wantsPlayback || !video.paused || state.starting) return;
  state.starting = true;
  video.play().then(() => {
    state.retry.hidden = true;
    if (document.hidden || (!state.visible && document.fullscreenElement !== video)) suspend(video);
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
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'video-play-fallback';
  retry.textContent = 'Play video';
  retry.setAttribute('aria-label', `Play ${video.getAttribute('aria-label') || 'video'}`);
  retry.hidden = true;
  video.after(retry);
  const state = { visible: false, wantsPlayback: !reducedMotion.matches, automaticPauses: 0, starting: false, retry };
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

const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    states.get(entry.target).visible = entry.isIntersecting;
    synchronize(entry.target);
  }
}, { threshold: 0 });
videos.forEach(video => observer.observe(video));
const synchronizeAll = () => videos.forEach(synchronize);
document.addEventListener('visibilitychange', synchronizeAll);
document.addEventListener('fullscreenchange', synchronizeAll);
window.addEventListener('pageshow', synchronizeAll);
window.addEventListener('focus', synchronizeAll);
window.addEventListener('pagehide', () => videos.forEach(suspend));
