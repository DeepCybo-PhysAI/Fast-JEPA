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

const pushtDriftAnchors = [[379.914,501.415],[379.6,501.724],[379.357,501.9],[379.114,502.075],[378.871,502.251],[378.627,502.426],[378.384,502.602],[378.141,502.777],[377.897,502.953],[377.654,503.128],[377.411,503.304],[377.168,503.479],[376.924,503.655],[376.681,503.831],[376.438,504.006],[376.195,504.182],[375.951,504.357],[375.708,504.533],[375.465,504.708],[375.221,504.884],[374.978,505.059],[374.735,505.235],[374.492,505.41],[370.452,502.011],[361.759,490.272],[357.659,481.151],[357.108,479.418],[356.867,479.588],[356.623,479.764],[356.38,479.94],[356.137,480.115],[355.893,480.291],[355.65,480.466],[355.407,480.642],[355.164,480.817],[354.92,480.993],[354.677,481.168],[354.434,481.344],[354.191,481.519],[353.947,481.695],[353.704,481.871],[353.461,482.046],[353.217,482.222],[352.974,482.397],[352.731,482.573],[352.488,482.748],[352.244,482.924],[352.001,483.099],[351.758,483.275],[352.155,481.704],[352.407,480.482],[352.714,480.124],[352.957,479.949],[353.2,479.773],[353.444,479.598],[353.687,479.422],[353.93,479.246],[354.173,479.071],[354.417,478.895],[354.66,478.72],[354.903,478.544],[355.146,478.369],[355.39,478.193],[355.633,478.018],[355.876,477.842],[356.12,477.667],[356.363,477.491],[356.606,477.315],[356.849,477.14],[357.093,476.964],[357.336,476.789],[357.579,476.613],[357.822,476.438],[358.066,476.262],[358.309,476.087],[358.552,475.911],[358.796,475.736],[359.039,475.56],[359.282,475.384],[359.525,475.209],[359.769,475.033],[360.012,474.858],[360.255,474.682],[360.499,474.507],[360.742,474.331],[360.985,474.156],[361.228,473.98],[361.472,473.804],[361.715,473.629],[361.958,473.453],[361.8,472.955],[361.922,472.682],[362.165,472.506],[361.91,471.928],[360.876,470.714],[360.791,470.269],[361.034,470.094],[360.882,469.593],[360.357,468.783],[360.226,468.296],[359.772,467.54]];
const pushtDriftDirection = [-0.810919147029477, 0.5851582153409327];

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
      card.querySelector('.goal-label text').textContent = after ? 'GOAL B' : 'GOAL A';
    } else if (kind === 'pusht') {
      status.textContent = after ? 'DRIFT REVERSED ↗' : 'External drift ↙';
      const frame = Math.min(pushtDriftAnchors.length - 1, Math.round(t * fps));
      const [x, y] = pushtDriftAnchors[frame];
      const angle = Math.atan2(pushtDriftDirection[1], pushtDriftDirection[0]) * 180 / Math.PI + (after ? 180 : 0);
      card.querySelector('.drift-arrow').setAttribute('transform', `translate(${x} ${y})`);
      card.querySelector('.drift-direction').setAttribute('transform', `rotate(${angle})`);
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
    const totalWait = Math.round((plan.ready_step - plan.observation_step) * schedule.dt * 1000);
    const elapsedWait = totalWait - left;
    const waitProgress = card.querySelector('.delay-wait-progress');
    waitProgress.querySelector('span').style.width = `${totalWait ? elapsedWait / totalWait * 100 : 100}%`;
    waitProgress.setAttribute('aria-valuemax', totalWait);
    waitProgress.setAttribute('aria-valuenow', elapsedWait);
    waitProgress.setAttribute('aria-valuetext', pending ? `${left} milliseconds remaining` : 'New plan ready');
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
    scene.textContent = `${event} · ${pending ? 'WAIT' : 'ACTIVE'}`;
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
