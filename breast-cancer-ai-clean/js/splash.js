const splashPage = document.getElementById('splashPage');

function goToHome() {
  window.location.href = 'home.html';
}

if (splashPage) {
  splashPage.addEventListener('click', goToHome);
}

setTimeout(() => {
  if (splashPage) splashPage.classList.add('fade-out');
}, 1700);

setTimeout(() => {
  goToHome();
}, 2200);
