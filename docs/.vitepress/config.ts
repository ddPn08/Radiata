import { defineConfig } from 'vitepress'

export default defineConfig({
    title: 'Radiata documentation',
    description: 'StableDiffusionWebUI using high-speed inference technology with TensorRT',
    lang: 'en-US',
    appearance: 'dark',
    lastUpdated: true,
    base: '/Radiata/',
    locales: {
        en: {
            label: 'English',
            lang: 'en',
            link: '/en/',
            themeConfig: {
                sidebar: [
                    {
                        text: 'Introduction',
                        items: [{ text: 'Introduction', link: '/en/' }],
                    },
                    {
                        text: 'Installation',
                        items: [
                            { text: 'Windows', link: '/en/installation/windows' },
                            { text: 'Linux', link: '/en/installation/linux' },
                        ],
                    },
                    {
                        text: 'Usage',
                        items: [
                            { text: 'WebUI', link: '/en/usage/' },
                            { text: 'Model', link: '/en/usage/model' },
                            { text: 'TensorRT', link: '/en/usage/tensorrt' },
                        ],
                    },
                    {
                        text: 'Developers',
                        items: [
                            { text: 'TensorRT', link: '/en/developers/tensorrt' },
                            {
                                text: 'Documentation',
                                link: '/en/developers/documentation',
                            },
                        ],
                    },
                    {
                        text: 'Troubleshooting',
                        items: [
                            { text: 'Linux', link: '/en/troubleshooting/linux' },
                            {
                                text: 'Windows',
                                link: '/en/troubleshooting/windows',
                            },
                        ],
                    },
                ],
            },
        },
        ja: {
            label: 'Japanese',
            lang: 'ja',
            link: '/ja/',
            themeConfig: {
                sidebar: [
                    {
                        text: '導入',
                        items: [{ text: 'Introduction', link: '/ja/' }],
                    },
                    {
                        text: 'インストール',
                        items: [
                            { text: 'Windows', link: '/ja/installation/windows' },
                            { text: 'Linux', link: '/ja/installation/linux' },
                        ],
                    },
                    {
                        text: '使い方',
                        items: [
                            { text: 'WebUI', link: '/ja/usage/' },
                            { text: 'Model', link: '/ja/usage/model' },
                            { text: 'TensorRT', link: '/ja/usage/tensorrt' },
                        ],
                    },
                    {
                        text: '開発者向け',
                        items: [
                            { text: 'TensorRT', link: '/ja/developers/tensorrt' },
                            {
                                text: 'Documentation',
                                link: '/ja/developers/documentation',
                            },
                        ],
                    },
                    {
                        text: 'トラブルシューティング',
                        items: [
                            { text: 'Linux', link: '/ja/troubleshooting/linux' },
                            {
                                text: 'Windows',
                                link: '/ja/troubleshooting/windows',
                            },
                        ],
                    },
                ],
            },
        },
    },
    themeConfig: {
        editLink: {
            pattern: 'https://github.com/ddPn08/Radiata/edit/main/docs/:path',
        },
        socialLinks: [
            {
                icon: 'github',
                link: 'https://github.com/ddPn08/Radiata',
            },
        ],
        sidebar: [
            {
                text: 'Radiata documentation',
                items: [{ text: 'Languages', link: '/' }],
            },
            {
                text: 'Languages',
                items: [
                    { text: 'English', link: '/en/' },
                    { text: '日本語', link: '/ja/' },
                ],
            },
        ],
        algolia: {
            appId: 'H0ENKDAOQE',
            indexName: 'lsmith',
            apiKey: '292dd349b89a56475ea02a16d4dac69b',
        },
    },
})
