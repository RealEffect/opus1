/* Page chrome: the signal-path rail, the "now reading" label (which opens the rail as a menu on narrow screens),
 * the progress bar, the language and theme buttons, in-page links and the data footer. */
(function () {
  'use strict';
  const secs = [...document.querySelectorAll('main > section[data-k]')];
  const subs = [...document.querySelectorAll('main h3[data-sub]')];
  const links = new Map([...document.querySelectorAll('.rail a')].map((a) => [a.getAttribute('href').slice(1), a]));
  const now = document.getElementById('now');
  const nowText = document.getElementById('now-text');
  const rail = document.getElementById('rail');
  const bar = document.getElementById('progress');
  const en = () => window.OX && OX.lang === 'en';

  function current(list) {
    const y = window.innerHeight * 0.3;
    let cur = null;
    for (const s of list) if (s.getBoundingClientRect().top <= y) cur = s;
    return cur;
  }
  let lastKey = null;
  function onScroll() {
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    bar.style.width = (max > 0 ? (window.scrollY / max) * 100 : 0) + '%';
    const s = current(secs) || secs[0];
    if (!s) return;
    // a sub-topic counts only while it is inside the current section
    let sub = current(subs.filter((h) => s.contains(h)));
    const key = s.id + '/' + (sub ? sub.id : '');
    if (key === lastKey) return;
    lastKey = key;
    nowText.innerHTML = '';
    const k = document.createElement('span');
    k.className = 'k';
    k.textContent = s.dataset.k;
    const title = (en() && s.dataset.titleEn) || s.dataset.title || '';
    const subTitle = sub ? (en() && sub.dataset.subEn) || sub.dataset.sub : '';
    nowText.append(k, title + (subTitle ? ' · ' + subTitle : ''));
    const active = sub && links.has(sub.id) ? sub.id : s.id;
    let passed = true;
    for (const [id, a] of links) {
      const isActive = id === active;
      const isParent = id === s.id && active !== s.id; // the section that holds the active sub-topic
      if (isActive) passed = false;
      a.classList.toggle('on', isActive || isParent);
      a.classList.toggle('done', links.has(s.id) && passed && !isParent);
      if (isActive) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    }
  }
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; onScroll(); });
  }, { passive: true });
  window.addEventListener('resize', onScroll);
  onScroll();

  /* On narrow screens the "now reading" label opens the rail as a menu. */
  const wide = window.matchMedia('(min-width: 1180px)');
  function setMenu(open) {
    rail.classList.toggle('open', open);
    now.setAttribute('aria-expanded', open ? 'true' : 'false');
  }
  now.addEventListener('click', () => { if (!wide.matches) setMenu(!rail.classList.contains('open')); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && rail.classList.contains('open')) { setMenu(false); now.focus(); } });
  document.addEventListener('click', (e) => {
    if (rail.classList.contains('open') && !rail.contains(e.target) && !now.contains(e.target)) setMenu(false);
  });

  /* In-page links: build the figures above the target first, so the jump lands on the final layout. */
  document.addEventListener('click', (e) => {
    const a = e.target.closest('a[href^="#"]');
    if (!a) return;
    const t = document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
    if (t && window.OX && OX.initBefore) OX.initBefore(t);
    if (rail.contains(a)) setMenu(false);
  });

  /* Theme: system -> light -> dark -> system. The saved choice was already applied by the script in <head>. */
  const btn = document.getElementById('theme-btn');
  const label = document.getElementById('theme-label');
  const NAMES = {
    zh: { system: '跟随系统', light: '浅色', dark: '深色' },
    en: { system: 'System', light: 'Light', dark: 'Dark' },
  };
  let mode = 'system';
  try { mode = localStorage.getItem('ox-theme') || 'system'; } catch (e) { /* storage blocked */ }
  if (!NAMES.zh[mode]) mode = 'system';
  function apply() {
    if (mode === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', mode);
    label.textContent = NAMES[en() ? 'en' : 'zh'][mode];
  }
  apply();
  btn.addEventListener('click', () => {
    mode = mode === 'system' ? 'light' : mode === 'light' ? 'dark' : 'system';
    try { localStorage.setItem('ox-theme', mode); } catch (e) { /* storage blocked */ }
    apply();
  });

  /* Language: the button names the other language. The choice was already applied by the script in <head>. */
  const langBtn = document.getElementById('lang-btn');
  function applyLangBtn() {
    langBtn.textContent = en() ? '中文' : 'EN';
    langBtn.setAttribute('aria-label', en() ? '切换到中文' : 'Switch to English');
  }
  applyLangBtn();
  langBtn.addEventListener('click', () => OX.setLang(en() ? 'zh' : 'en'));

  /* Footer: where the real data came from. The data script loads last, so wait for it. */
  function footer() {
    const meta = document.getElementById('data-meta');
    const D = window.OPUS_DEMO_DATA;
    if (!meta) return;
    const t = (zh, enText) => (en() ? enText : zh);
    if (!D) {
      meta.textContent = t('没有找到 data/opus-demo-data.js，标“真实数据”的图无法显示。',
        'data/opus-demo-data.js was not found, so the figures marked “Real data” cannot be shown.');
      return;
    }
    const m = D.meta || {};
    const parts = [];
    if (m.crate) parts.push(m.crate + (m.crate_version ? ' ' + m.crate_version : ''));
    if (m.git_commit && m.git_commit !== 'unknown') parts.push(t('提交 ', 'commit ') + m.git_commit);
    if (m.libopus_version) parts.push(t('对照参考：', 'reference: ') + m.libopus_version);
    if (m.generator) parts.push(t('生成程序：', 'generator: ') + m.generator);
    meta.textContent = parts.join(' · ');
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', footer);
  else footer();

  OX.onLang(() => {
    lastKey = null;
    onScroll();
    apply();
    applyLangBtn();
    footer();
  });
})();
