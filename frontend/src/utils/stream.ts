export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    if (!reader) return
    while (true) {
        const res = await reader.read()
        if (res.done) return
        try {
            const text = new TextDecoder().decode(res.value).split('\n')
            yield JSON.parse(text[text.length - 2])
        } catch (e) {
            console.error(e)
        }
    }
}
