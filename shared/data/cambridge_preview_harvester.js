(() => {
  // Diagnostics first so we can see what's actually on the page.
  const audioCount = document.querySelectorAll("audio").length;
  const sourceCount = document.querySelectorAll("audio source").length;
  console.log("audio tags: " + audioCount + "   source tags inside audio: " + sourceCount);
  if (audioCount) {
    const sample = document.querySelector("audio");
    console.log("first <audio> outerHTML:\n" + sample.outerHTML);
  }

  // Robust harvest: scan the entire rendered HTML for *_Full_Preview.mp3 URLs.
  // This catches src= values regardless of whether they're on <audio>, <source>,
  // <a>, data-* attributes, inline JS strings, etc.
  const html = document.documentElement.outerHTML;
  const re = /https?:\/\/[^"'<>\s]+?_Full_Preview\.mp3/gi;
  const urls = [...new Set(html.match(re) || [])];
  console.log("Full_Preview.mp3 URLs found in raw HTML: " + urls.length);
  console.log(urls.slice(0, 10).join("\n"));

  const text = urls.join("\n") + "\n";
  copy(text);
  console.log("copied " + urls.length + " URLs to clipboard via DevTools copy()");
  return urls;
})();
