<template>
  <router-view />
</template>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Lato', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #FAF5FF;
  color: #1E1B4B;
  height: 100vh;
}
#app { height: 100vh; }

:root {
  --color-primary: #7C3AED;
  --color-primary-dark: #5B21B6;
  --color-primary-light: #EDE9FE;
  --color-surface: #FFFFFF;
  --color-bg: #FAF5FF;
  --color-border: #DDD6FE;
  --color-text: #1E1B4B;
  --color-text-muted: #6B7280;
  --color-error: #DC2626;
  --font-heading: 'EB Garamond', serif;
  --font-body: 'Lato', -apple-system, sans-serif;
  --radius: 8px;
  --radius-lg: 12px;
  --transition: 150ms ease;
}

h1, h2, h3 { font-family: var(--font-heading); }
</style>
