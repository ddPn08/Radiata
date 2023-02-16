export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    const placeholder = '{placeholder}'
    if (!reader) return
    while (true) {
        const res = await reader.read()
        if (res.done) return
        try {
            const text = new TextDecoder()
                .decode(res.value)
                .replaceAll('}{', `}${placeholder}{`)
                .split(placeholder)
            yield JSON.parse(text[text.length - 1])
        } catch (_) {}
    }
}
