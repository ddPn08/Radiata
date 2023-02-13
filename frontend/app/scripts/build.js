import dotenv from 'dotenv'
import esbuild from 'esbuild'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

import packageJson from '../package.json' assert { type: 'json' }

dotenv.config()

const __dirname = fileURLToPath(path.dirname(import.meta.url))
const __dev = process.env['NODE_ENV'] === 'development'
const cwd = path.dirname(__dirname)

/** @type {import('esbuild').Plugin} */
export const ExternalExporter = {
  name: 'lsmith:external-exporter',
  setup(build) {
    const filter = new RegExp(path.join(cwd, 'src', 'main.tsx').replaceAll('\\', '\\\\'))
    build.onLoad({ filter }, async (args) => {
      let contents = await fs.promises.readFile(args.path, 'utf8')
      if (contents.includes('/** @LOAD_GLOBALS */')) {
        const externals = Object.keys(packageJson.dependencies)
          .map((v) => `window['__LSMITH_GLOBALS']['${v}']=await import('${v}')`)
          .join(';')
        contents = contents.replace('/** @LOAD_GLOBALS */', `${externals}`)
      }
      return {
        pluginData: {
          contents,
        },
        contents,
        loader: 'tsx',
      }
    })
  },
}

/**
 * @param {boolean} watch
 */
const bundle = async (watch) => {
  const outdir = path.join(cwd, 'js')
  if (fs.existsSync(outdir)) await fs.promises.rm(outdir, { recursive: true })

  /** @type {Record<string, any>} */
  const define = {}

  for (const k in process.env) {
    if (k.startsWith('REACT_APP_')) {
      define[`process.env.${k}`] = JSON.stringify(process.env[k])
    }
  }

  /** @type {import('esbuild').BuildOptions} */
  const options = {
    logLevel: 'info',
    bundle: true,
    sourcemap: __dev,
    minify: !__dev,
    entryPoints: [path.join(cwd, 'src', 'main.tsx')],
    outfile: path.join(outdir, 'index.js'),
    format: 'esm',
    platform: 'browser',
    plugins: [ExternalExporter],
    jsxFactory: 'jsx',
    jsxFragment: 'Fragment',
    inject: [path.join(__dirname, 'shims', 'react.js')],
    define: define,
  }

  if (watch) {
    const ctx = await esbuild.context(options)
    await ctx.watch()
  } else {
    await esbuild.build(options)
  }
}

const watch = process.argv.includes('--watch')

await bundle(watch)
