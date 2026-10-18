// Language state shared by the host (index.html) and every scene module. Vietnamese is the default.
// Choice order: ?lang=vi|en in the URL, then the last choice saved in localStorage, then 'vi'.
// See js/scenes/CONTRACT.md ("Language") for how scenes provide English text.
export const LANGS = ['vi', 'en'];
const KEY = 'interstellar.lang';

function detect() {
  try {
    const q = new URLSearchParams(location.search).get('lang');
    if (LANGS.includes(q)) return q;
  } catch (e) { /* no location (worker) */ }
  try {
    const s = localStorage.getItem(KEY);
    if (LANGS.includes(s)) return s;
  } catch (e) { /* storage blocked */ }
  return 'vi';
}

let cur = detect();
const listeners = [];

/** current language: 'vi' | 'en' */
export const lang = () => cur;
/** pick the string for the current language (falls back to Vietnamese when `en` is missing) */
export const t = (vi, en) => (cur === 'en' && en != null ? en : vi);
/** BCP 47 locale for toLocaleString() */
export const locale = () => (cur === 'en' ? 'en-US' : 'vi-VN');
/** decimal with the language's separator: VI 3,14 · EN 3.14 (no grouping) */
export const dec = (x, d = 1) => (cur === 'en' ? x.toFixed(d) : x.toFixed(d).replace('.', ','));
/** rounded integer with thousands grouping: VI 1 234 567 · EN 1,234,567 */
export const int = n => Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, cur === 'en' ? ',' : ' ');

/** English override of a static text field: obj.en[field] in English, obj[field] otherwise */
export const tr = (obj, field) => (obj && cur === 'en' && obj.en && obj.en[field] != null ? obj.en[field] : obj && obj[field]);

export function setLang(l, { persist = true } = {}) {
  if (!LANGS.includes(l) || l === cur) return false;
  cur = l;
  if (persist) { try { localStorage.setItem(KEY, l); } catch (e) { /* storage blocked */ } }
  for (const f of listeners) { try { f(l); } catch (e) { console.error('[i18n]', e); } }
  return true;
}
/** subscribe to language changes */
export const onLang = f => { listeners.push(f); };
