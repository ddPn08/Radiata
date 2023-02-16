export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    if (!reader) return
    while (true) {
        const res = await reader.read()
        if (res.done) return
        try {
            yield JSON.parse(new TextDecoder().decode(res.value) || '')
        } catch (_) {}
    }
}
