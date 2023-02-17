import { defineConfig } from "vitepress";

export default defineConfig({
	title: "Lsmith documentation",
	description:
		"StableDiffusionWebUI using high-speed inference technology with TensorRT",
	lang: "en-US",
	appearance: "dark",
	lastUpdated: true,
	base: "/Lsmith/",
	themeConfig: {
		editLink: {
			pattern: "https://github.com/ddPn08/Lsmith/edit/main/docs/:path",
		},
		socialLinks: [
			{
				icon: "github",
				link: "https://github.com/ddPn08/Lsmith",
			},
		],
		sidebar: [
			{
				text: "Introduction",
				items: [{ text: "Introduction", link: "/" }],
			},
			{
				text: "Installation",
				items: [
					{ text: "Docker", link: "/installation/docker" },
					{ text: "Windows", link: "/installation/windows" },
					{ text: "Linux", link: "/installation/linux" },
				],
			},
			{
				text: "WebUI",
				items: [{ text: "WebUI", link: "/webui/" }],
			},
			{
				text: "API",
				items: [{ text: "API", link: "/api/" }],
			},
			{
				text: "Developers",
				items: [
					{ text: "TensorRT", link: "/developers/tensorrt" },
					{
						text: "Frontend",
						link: "/developers/frontend",
					},
					{
						text: "Documentation",
						link: "/developers/documentation",
					},
				],
			},
			{
				text: "Troubleshooting",
				items: [
					{ text: "Linux", link: "/troubleshooting/linux" },
					{
						text: "Windows",
						link: "/troubleshooting/windows",
					},
					{
						text: "Docker",
						link: "/troubleshooting/docker",
					},
				],
			},
		],
	},
});
