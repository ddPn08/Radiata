import react from '@vitejs/plugin-react-swc'
import path from 'path'
import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
    base: '/app',
    // build: { target: 'esnext' },
    resolve: {
        alias: {
            '~': path.resolve(__dirname, 'src'),
            'internal:api': path.resolve(__dirname, 'modules', 'api'),
        },
    },
    plugins: [react()],
})
