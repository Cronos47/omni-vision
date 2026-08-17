/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        panel: '#0f1218',
        borderSubtle: 'rgba(255,255,255,0.08)'
      },
      boxShadow: {
        glow: '0 0 80px rgba(34,211,238,0.16)',
      },
      backgroundImage: {
        mesh: 'radial-gradient(circle at top left, rgba(34,211,238,0.18), transparent 35%), radial-gradient(circle at top right, rgba(217,70,239,0.16), transparent 32%), radial-gradient(circle at bottom center, rgba(168,85,247,0.15), transparent 40%)'
      }
    },
  },
  plugins: [],
}
