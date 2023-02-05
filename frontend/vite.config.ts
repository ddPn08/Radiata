import path from 'path'
import Icons from 'unplugin-icons/vite'
import { defineConfig } from 'vite'
import solidPlugin from 'vite-plugin-solid'

export default defineConfig({
    base: '/app',
    build: { target: 'esnext' },
    resolve: {
        alias: {
            '~': path.resolve(__dirname, 'src'),
            'internal:api': path.resolve(__dirname, 'modules', 'api'),
        },
    },
    plugins: [
        Icons({
            compiler: 'solid',
        }),
        solidPlugin(),
    ],
})
