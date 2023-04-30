import { defineConfig } from 'vitepress'

export default defineConfig({
    title: 'Radiata documentation',
    description: 'StableDiffusionWebUI using high-speed inference technology with TensorRT',
    lang: 'en-US',
    appearance: 'dark',
    lastUpdated: true,
    base: '/Radiata/',
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
                text: 'Introduction',
                items: [{ text: 'Introduction', link: '/' }],
            },
            {
                text: 'Installation',
                items: [
                    { text: 'Windows', link: '/installation/windows' },
                    { text: 'Linux', link: '/installation/linux' },
                ],
            },
            {
                text: 'WebUI',
                items: [
                    { text: 'WebUI', link: '/webui/' },
                    { text: 'Model', link: '/webui/model' },
                ],
            },
            {
                text: 'Developers',
                items: [
                    { text: 'TensorRT', link: '/developers/tensorrt' },
                    {
                        text: 'Documentation',
                        link: '/developers/documentation',
                    },
                ],
            },
            {
                text: 'Troubleshooting',
                items: [
                    { text: 'Linux', link: '/troubleshooting/linux' },
                    {
                        text: 'Windows',
                        link: '/troubleshooting/windows',
                    },
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
