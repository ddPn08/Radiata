import cac from 'cac'
import dotenv from 'dotenv'

import * as builder from './builder'
import packageJson from '../../../package.json'

export type BaseCliContext = {
    args: {}
}

dotenv.config()

const cli = cac('lsmith')

cli.version(packageJson.version)

cli.command('build').action(() => {
    builder.run()
})

cli.help()
cli.parse()
