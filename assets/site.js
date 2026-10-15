// Email links. The address is not written anywhere in the page: it is kept reversed in an
// attribute, and put together only when a person points at, focuses or clicks the link.
(function () {
  function address(a) { return a.dataset.e.split('').reverse().join(''); }
  function arm(a) { a.href = 'mailto:' + address(a); }

  document.querySelectorAll('a.eml').forEach(function (a) {
    ['pointerenter', 'focus', 'touchstart'].forEach(function (ev) {
      a.addEventListener(ev, function () { arm(a); }, { once: true, passive: true });
    });
    a.addEventListener('click', function () { arm(a); });
  });

  // The CV shows the address in full on paper and in the PDF, and only there.
  var printed = document.querySelectorAll('a.eml[data-print]');
  function reveal() { printed.forEach(function (a) { a.dataset.label = a.textContent; a.textContent = address(a); }); }
  function hide() { printed.forEach(function (a) { if (a.dataset.label) a.textContent = a.dataset.label; }); }
  window.addEventListener('beforeprint', reveal);
  window.addEventListener('afterprint', hide);
  if (/[?&]print\b/.test(location.search)) reveal();
})();
