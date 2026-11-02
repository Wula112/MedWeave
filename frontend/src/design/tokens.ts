/**
 * Design tokens: the single source for `tailwind.config.ts` and for the few
 * places that need raw values at runtime (the voice orb's gradients).
 *
 * The voice call screen is the first consumer; the rest of the app moves onto
 * these tokens in later rounds. Use the semantic names (`brand`, `ink`, …),
 * never a raw Tailwind colour family, in code that has adopted them.
 */

/** Clear blue: primary actions, listening state, and filled facts. */
const brand = {
    50: '#eff5ff', 100: '#dceaff', 200: '#c2d9ff', 300: '#9abfff', 400: '#77a5f8',
    500: '#4781f2', 600: '#326be8', 700: '#2d64e5', 800: '#214db4', 900: '#193b7c',
} as const;

/** Blue-grey: text, surfaces, and hairlines. 500 is the lightest readable text. */
const ink = {
    50: '#f3f7fc', 100: '#eaf0f8', 200: '#dbe4f0', 300: '#c6d2e2', 400: '#91a3bb',
    500: '#5d6f89', 600: '#4d5e77', 700: '#34465f', 800: '#22334d', 900: '#16263d',
} as const;

/** Violet: the one accent, reserved for "the assistant is thinking". */
const accent = {
    50: '#f7f3ff', 100: '#efe7ff', 200: '#dfd0fa', 300: '#c9b1f2', 400: '#b28de9',
    500: '#9871db', 600: '#8059c7', 700: '#6a43b4', 800: '#573394',
} as const;

/** Sky: system information and "the assistant is speaking" (audio playback). */
const info = {
    50: '#eff8ff', 100: '#dff2ff', 200: '#bde6fc', 300: '#86d0f4', 400: '#55b8eb',
    500: '#299bd8', 600: '#087eb5', 700: '#076493', 800: '#0a4f76',
} as const;

/** Mint: reassuring, positive status. */
const success = { 50: '#eafaf4', 100: '#d3f4e9', 200: '#aee8d6', 500: '#28b997', 600: '#0d8570', 700: '#087363', 800: '#075b50' } as const;

/** Amber: needs attention but not dangerous (confirm, push-to-talk, missing facts). */
const warning = {
    50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d', 400: '#fbbf24',
    500: '#f59e0b', 600: '#d97706', 700: '#b45309', 800: '#92400e',
} as const;

/** Coral: hang-up, emergency, errors only. */
const danger = {
    50: '#fff2f4', 100: '#ffe4e9', 200: '#fecdd7', 300: '#fba9b8', 400: '#f88191',
    500: '#ed5b72', 600: '#d83f59', 700: '#b92b47', 800: '#93213a',
} as const;

export const colors = { brand, ink, accent, info, success, warning, danger } as const;

/**
 * Type scale (size / line height). Nothing below 12px; body copy is 14px+.
 * Elder mode moves text one step up.
 */
export const fontSize = {
    caption: ['12px', { lineHeight: '16px' }],
    body: ['14px', { lineHeight: '22px' }],
    callout: ['16px', { lineHeight: '24px' }],
    title: ['18px', { lineHeight: '26px' }],
    headline: ['22px', { lineHeight: '31px' }],
    display: ['28px', { lineHeight: '38px' }],
} satisfies Record<string, [string, { lineHeight: string }]>;

/** System Chinese fonts only; nothing is downloaded. */
export const fontFamily = {
    cn: ['"PingFang SC"', '"HarmonyOS Sans SC"', '"Noto Sans SC"', 'system-ui', 'sans-serif'],
};

/** card: cards and sheets; control: buttons and inputs; full (built in): chips and round buttons. */
export const borderRadius = {
    card: '24px',
    control: '16px',
} as const;

/**
 * raised: resting cards and buttons; floating: sheets, overlays, the primary
 * call-to-action; docked: surfaces pinned to the bottom edge (the shadow falls upward).
 */
export const boxShadow = {
    raised: '0 1px 3px rgba(38, 69, 113, 0.05), 0 8px 24px -12px rgba(52, 96, 159, 0.18)',
    floating: '0 2px 6px rgba(38, 69, 113, 0.06), 0 24px 48px -18px rgba(52, 96, 159, 0.27)',
    docked: '0 -1px 2px rgba(38, 69, 113, 0.04), 0 -12px 32px -16px rgba(52, 96, 159, 0.16)',
} as const;

/** fast: press feedback; base: colour/state changes; slow: sheets and entrances. */
export const transitionDuration = {
    fast: '150ms',
    base: '200ms',
    slow: '300ms',
} as const;
