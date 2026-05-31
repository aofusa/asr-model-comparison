// Quick diagnostic: load the prod build and report what actually rendered + any errors
import { chromium } from '@playwright/test';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const errors = [];
  const consoleMessages = [];

  page.on('pageerror', err => errors.push(err.message));
  page.on('console', msg => {
    if (msg.type() === 'error') consoleMessages.push(`[console.error] ${msg.text()}`);
  });

  await page.goto('http://localhost:8000', { waitUntil: 'networkidle' });
  await page.waitForTimeout(2000); // give Qwik time to hydrate

  const html = await page.content();
  const rootInner = await page.locator('#root').innerHTML().catch(() => '<not found>');
  const bodyText = (await page.textContent('body').catch(() => '')).trim().slice(0, 500);

  console.log('=== PAGE ERRORS ===');
  console.log(errors.length ? errors.join('\n') : '(none)');
  console.log('\n=== CONSOLE ERRORS ===');
  console.log(consoleMessages.length ? consoleMessages.join('\n') : '(none)');
  console.log('\n=== #root innerHTML (first 800 chars) ===');
  console.log(rootInner.slice(0, 800));
  console.log('\n=== body visible text (first 500 chars) ===');
  console.log(bodyText || '(empty)');

  await browser.close();
  process.exit(errors.length > 0 ? 1 : 0);
})();
