/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Syne"', 'sans-serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        void:    '#05050a',
        ink:     '#0d0d1a',
        surface: '#12121f',
        card:    '#1a1a2e',
        border:  '#252538',
        muted:   '#3a3a55',
        subtle:  '#6b6b8a',
        text:    '#e8e8f0',
        bright:  '#ffffff',
        plasma:  '#7c3aed',
        neon:    '#a78bfa',
        arc:     '#38bdf8',
        flare:   '#fb7185',
        amber:   '#f59e0b',
        lime:    '#84cc16',
      },
      backgroundImage: {
        'grid-pattern': `linear-gradient(rgba(124,58,237,0.05) 1px, transparent 1px),
                         linear-gradient(90deg, rgba(124,58,237,0.05) 1px, transparent 1px)`,
        'plasma-radial': 'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(124,58,237,0.25), transparent)',
        'arc-radial':    'radial-gradient(ellipse 60% 40% at 80% 60%, rgba(56,189,248,0.12), transparent)',
        'hero-mesh':     `
          radial-gradient(ellipse 100% 80% at 10% 20%, rgba(124,58,237,0.18) 0%, transparent 60%),
          radial-gradient(ellipse 80% 60% at 90% 80%, rgba(56,189,248,0.12) 0%, transparent 60%),
          radial-gradient(ellipse 60% 60% at 50% 50%, rgba(251,113,133,0.06) 0%, transparent 60%)
        `,
      },
      animation: {
        'float':       'float 6s ease-in-out infinite',
        'pulse-slow':  'pulse 4s ease-in-out infinite',
        'spin-slow':   'spin 20s linear infinite',
        'shimmer':     'shimmer 2s linear infinite',
        'scan':        'scan 3s ease-in-out infinite',
        'blink':       'blink 1.2s step-end infinite',
        'glow-pulse':  'glowPulse 2s ease-in-out infinite',
        'rise':        'rise 0.8s cubic-bezier(0.16,1,0.3,1) forwards',
      },
      keyframes: {
        float: {
          '0%,100%': { transform: 'translateY(0px)' },
          '50%':     { transform: 'translateY(-12px)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        scan: {
          '0%,100%': { opacity: '0.3', transform: 'scaleX(0.8)' },
          '50%':     { opacity: '1',   transform: 'scaleX(1)' },
        },
        blink: {
          '0%,100%': { opacity: '1' },
          '50%':     { opacity: '0' },
        },
        glowPulse: {
          '0%,100%': { boxShadow: '0 0 20px rgba(124,58,237,0.3)' },
          '50%':     { boxShadow: '0 0 40px rgba(124,58,237,0.6), 0 0 80px rgba(56,189,248,0.2)' },
        },
        rise: {
          from: { opacity: '0', transform: 'translateY(30px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
      backdropBlur: { xs: '2px' },
    },
  },
  plugins: [],
}
