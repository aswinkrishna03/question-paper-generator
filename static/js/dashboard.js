(function () {
  const toggleButton = document.querySelector('.site-header__menu-toggle');
  const navMenu = document.querySelector('.site-header__nav');

  if (!toggleButton || !navMenu) {
    return;
  }

  const mobileQuery = window.matchMedia('(max-width: 768px)');

  const isMenuOpen = () => toggleButton.getAttribute('aria-expanded') === 'true';

  const openMenu = () => {
    navMenu.hidden = false;
    toggleButton.setAttribute('aria-expanded', 'true');
    toggleButton.setAttribute('aria-label', 'Close navigation menu');
  };

  const closeMenu = () => {
    if (mobileQuery.matches) {
      navMenu.hidden = true;
    }
    toggleButton.setAttribute('aria-expanded', 'false');
    toggleButton.setAttribute('aria-label', 'Open navigation menu');
  };

  const syncMenuState = () => {
    if (mobileQuery.matches) {
      navMenu.hidden = !isMenuOpen();
    } else {
      navMenu.hidden = false;
      toggleButton.setAttribute('aria-expanded', 'false');
      toggleButton.setAttribute('aria-label', 'Open navigation menu');
    }
  };

  toggleButton.addEventListener('click', function () {
    if (isMenuOpen()) {
      closeMenu();
    } else {
      openMenu();
    }
  });

  navMenu.addEventListener('click', function (event) {
    if (event.target.classList.contains('site-header__nav-link') && mobileQuery.matches) {
      closeMenu();
    }
  });

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && isMenuOpen()) {
      closeMenu();
      toggleButton.focus();
    }
  });

  document.addEventListener('click', function (event) {
    if (!mobileQuery.matches || !isMenuOpen()) {
      return;
    }

    const clickedInsideHeader = event.target.closest('.site-header__container');
    if (!clickedInsideHeader) {
      closeMenu();
    }
  });

  syncMenuState();
  window.addEventListener('resize', syncMenuState);
})();
