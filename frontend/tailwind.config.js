/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js}'],
  theme: {
    extend: {
      colors: {
        // Apple 系统蓝作为主色（从原 indigo 切换）
        primary: {
          50:  '#f0f7ff',
          100: '#e0f0ff',
          200: '#b9ddff',
          300: '#7cc0ff',
          400: '#3aa0ff',
          500: '#0a84ff',
          600: '#0071e3',
          700: '#005bb8',
          800: '#004a94',
          900: '#003d78',
        },
        // Apple 中性灰阶（更冷、更克制）
        neutral: {
          50:  '#f5f5f7',
          100: '#e8e8ed',
          200: '#d2d2d7',
          300: '#aeaeb2',
          400: '#86868b',
          500: '#6e6e73',
          600: '#515154',
          700: '#3a3a3c',
          800: '#1d1d1f',
          900: '#0d0d0f',
        },
        // 状态色（Apple 系统色）
        success: '#34c759',
        warning: '#ff9f0a',
        danger:  '#ff3b30',
      },
      fontFamily: {
        // SF Pro 优先（Mac 原生），中文回退 PingFang SC
        sans: ['-apple-system', 'BlinkMacSystemFont', '"SF Pro Text"', '"SF Pro Display"',
          '"PingFang SC"', '"HarmonyOS Sans"', '"Microsoft YaHei"', 'system-ui', 'sans-serif'],
        mono: ['"SF Mono"', '"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        // Apple 风的多层柔和阴影
        soft: '0 1px 3px rgba(0, 0, 0, 0.04), 0 8px 24px rgba(0, 0, 0, 0.06)',
        card: '0 1px 2px rgba(0, 0, 0, 0.03), 0 4px 16px rgba(0, 0, 0, 0.05)',
        float: '0 4px 12px rgba(0, 0, 0, 0.08), 0 16px 48px rgba(0, 0, 0, 0.12)',
        'primary-glow': '0 4px 16px rgba(0, 113, 227, 0.28)',
      },
      borderRadius: {
        xl2: '1rem',
        xl3: '1.25rem',
      },
    },
  },
  plugins: [],
}
