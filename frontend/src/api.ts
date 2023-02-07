import { Configuration, MainApi } from 'internal:api'

const BASE_PATH = (import.meta.env['VITE_API_BASE_PATH'] as string) || ''

const config = {
    basePath: BASE_PATH,
}

if (!BASE_PATH.startsWith('http')) {
    config.basePath = `${window.location.origin}${BASE_PATH}`
}

export const api = new MainApi(new Configuration(config))

export const createUrl = (pathname: string) => `${BASE_PATH}${pathname}`
