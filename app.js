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

// Cues are driven by decoded video time, including native seeking and loops.
for (const card of document.querySelectorAll('[data-shift]')) {
  const video = card.querySelector('video');
  const kind = card.dataset.shift;
  const indicator = card.querySelector('.shift-indicator');
  const status = card.querySelector('[data-shift-status]');
  const clock = card.querySelector('[data-shift-clock]');
  const fill = card.querySelector('.shift-track-fill');
  const tick = card.querySelector('.shift-event-tick');
  const fps = {cube:20,reacher:25,pusht:10,tworoom:10}[kind];
  const eventTime = {cube:1.25,reacher:3,pusht:5.1,tworoom:5}[kind];
  function update(mediaTime = video.currentTime) {
    const t = Math.min(100 / fps, Math.floor((mediaTime + 0.00001) * fps) / fps);
    const after = t >= eventTime;
    clock.textContent = `${t.toFixed(2)} s`;
    fill.style.width = `${Math.min(100, mediaTime / (video.duration || 100 / fps) * 100)}%`;
    tick.style.left = `${eventTime / (video.duration || 101 / fps) * 100}%`;
    card.classList.toggle('is-shifted', after && kind !== 'tworoom');
    indicator.classList.toggle('is-event', after && t < eventTime + .8);
    indicator.dataset.phase = kind === 'tworoom' ? 'moving' : after ? 'after' : 'before';
    if (kind === 'cube' || kind === 'reacher') {
      status.textContent = after ? 'GOAL SWITCHED · A → B' : 'Goal A · switch ahead';
      card.querySelector('.goal-label text').textContent = after ? 'GOAL B · NEW TARGET' : 'GOAL A';
    } else if (kind === 'pusht') {
      status.textContent = after ? 'DRIFT REVERSED ↗' : 'External drift ↙';
      card.querySelector('.drift-arrow path').setAttribute('transform', after ? 'rotate(180)' : 'rotate(0)');
    } else {
      const offset = 8 * Math.sin(2 * Math.PI * t / 20);
      const down = t < 5;
      status.textContent = `Door ${down ? '↓' : '↑'} · offset +${Math.max(0,offset).toFixed(1)} px`;
      // Native doorway center 49, half-width 14, in a 224 px world.
      // The renderer maps each scene to 768 px with a 54 px top label.
      const y = 54 + (49 - 14 + offset) * 768 / 224;
      card.querySelectorAll('.door-highlight').forEach(rect => rect.setAttribute('y', y.toFixed(2)));
      card.querySelectorAll('.door-motion-arrow').forEach(arrow => arrow.setAttribute('transform', down ? '' : 'translate(0 440) scale(1 -1)'));
    }
  }
  update();
  for (const event of ['loadedmetadata','seeked','timeupdate','play','pause']) video.addEventListener(event, () => update());
  if ('requestVideoFrameCallback' in video) {
    const onFrame = (_, metadata) => { update(metadata.mediaTime); video.requestVideoFrameCallback(onFrame); };
    video.requestVideoFrameCallback(onFrame);
  }
}

const delaySchedules = {
  "cube": {
    "dt": 0.05,
    "plans": [
      {
        "observation_step": 0,
        "ready_step": 16
      },
      {
        "observation_step": 25,
        "ready_step": 41
      },
      {
        "observation_step": 50,
        "ready_step": 66
      },
      {
        "observation_step": 75,
        "ready_step": 91
      }
    ],
    "sources": [
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan"
    ]
  },
  "tworoom": {
    "dt": 0.1,
    "plans": [
      {
        "observation_step": 0,
        "ready_step": 8
      },
      {
        "observation_step": 25,
        "ready_step": 33
      },
      {
        "observation_step": 50,
        "ready_step": 58
      },
      {
        "observation_step": 75,
        "ready_step": 83
      }
    ],
    "sources": [
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan"
    ]
  },
  "pusht": {
    "dt": 0.1,
    "plans": [
      {
        "observation_step": 0,
        "ready_step": 8
      },
      {
        "observation_step": 25,
        "ready_step": 33
      },
      {
        "observation_step": 50,
        "ready_step": 58
      },
      {
        "observation_step": 75,
        "ready_step": 83
      }
    ],
    "sources": [
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan"
    ]
  },
  "reacher": {
    "dt": 0.04,
    "plans": [
      {
        "observation_step": 0,
        "ready_step": 20
      },
      {
        "observation_step": 25,
        "ready_step": 45
      },
      {
        "observation_step": 50,
        "ready_step": 70
      },
      {
        "observation_step": 75,
        "ready_step": 95
      }
    ],
    "sources": [
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "hold",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan",
      "plan"
    ]
  }
};

// Recorded observation / ready steps define the imposed response delay.
for (const card of document.querySelectorAll('[data-delay]')) {
  const task = card.dataset.delay;
  const schedule = delaySchedules[task];
  const video = card.querySelector('video');
  const indicator = card.querySelector('.delay-indicator');
  const status = card.querySelector('[data-delay-status]');
  const count = card.querySelector('[data-delay-count]');
  const action = card.querySelector('[data-delay-action]');
  const scene = card.querySelector('[data-delay-scene]');
  function update(mediaTime = video.currentTime) {
    const step = Math.min(100, Math.floor((mediaTime + .00001) / schedule.dt));
    let plan = schedule.plans[0];
    for (const next of schedule.plans) if (next.observation_step <= step) plan = next;
    const pending = step < plan.ready_step;
    const left = Math.max(0, Math.round((plan.ready_step - step) * schedule.dt * 1000));
    const t = step * schedule.dt;
    indicator.dataset.phase = pending ? 'pending' : 'active';
    card.classList.toggle('plan-pending', pending);
    indicator.classList.toggle('is-event', !pending && t < plan.ready_step * schedule.dt + .4);
    status.textContent = pending ? 'NEW PLAN PENDING' : 'NEW PLAN ACTIVE';
    count.textContent = pending ? `${left} ms` : 'READY ✓';
    const source = schedule.sources[Math.min(step, schedule.sources.length - 1)];
    action.textContent = pending
      ? source === 'hold' ? 'Held action continues; environment keeps evolving.' : 'Previous plan continues; new plan is not ready yet.'
      : 'Executing the newly available action plan.';
    card.querySelector('[data-delay-stage="wait"]').classList.toggle('active', pending);
    card.querySelector('[data-delay-stage="apply"]').classList.toggle('active', !pending);
    let event = task === 'cube' ? (step >= 25 ? 'GOAL B' : 'GOAL A')
      : task === 'reacher' ? (step >= 75 ? 'GOAL B' : 'GOAL A')
      : task === 'pusht' ? (step > 50 ? 'REVERSED DRIFT' : 'EXTERNAL DRIFT')
      : 'MOVING DOOR';
    scene.textContent = `${event} · ${pending ? 'PLAN PENDING' : 'PLAN ACTIVE'}`;
    card.querySelector('[data-delay-observed]').textContent = `Observed ${(plan.observation_step * schedule.dt).toFixed(2)} s`;
    card.querySelector('[data-delay-ready]').textContent = `Plan ready ${(plan.ready_step * schedule.dt).toFixed(2)} s`;
    card.querySelector('.delay-track span').style.width = `${Math.min(100, (step - plan.observation_step) / 25 * 100)}%`;
    card.querySelector('.delay-track i').style.left = `${(plan.ready_step - plan.observation_step) / 25 * 100}%`;
  }
  update();
  for (const event of ['loadedmetadata','seeked','timeupdate','play','pause']) video.addEventListener(event, () => update());
  if ('requestVideoFrameCallback' in video) {
    const onFrame = (_, metadata) => { update(metadata.mediaTime); video.requestVideoFrameCallback(onFrame); };
    video.requestVideoFrameCallback(onFrame);
  }
}
