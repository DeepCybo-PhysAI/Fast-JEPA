'use strict';
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const videos=[...document.querySelectorAll('video')];
// Only visible videos play; native controls remain available for every clip.
const observer=new IntersectionObserver(entries=>{for(const e of entries){const v=e.target;if(e.isIntersecting&&!reduced&&!document.hidden){v.play().catch(()=>{});}else{v.pause();}}},{threshold:.45});
videos.forEach(v=>observer.observe(v));
document.addEventListener('visibilitychange',()=>{if(document.hidden)videos.forEach(v=>v.pause());});
