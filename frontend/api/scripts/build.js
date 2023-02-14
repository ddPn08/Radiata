import fs from 'fs'
import path from 'path'
import tsup from 'tsup'
import { fileURLToPath } from 'url'

const __dirname = fileURLToPath(path.dirname(import.meta.url))

if (fs.existsSync('./dist')) fs.rmSync('./dist', { recursive: true })

await tsup.build({
    entry: { cli: path.join(__dirname, '..', 'src', 'cli', 'index.ts') },
    format: 'cjs',
    platform: 'node',
    bundle: true,
})
await tsup.build({
    entry: { index: path.join(__dirname, '..', 'src', 'index.ts') },
    dts: true,
    format: ['esm', 'cjs'],
    platform: 'node',
    bundle: true,
})
