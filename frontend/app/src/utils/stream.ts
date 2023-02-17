export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    const placeholder = '{placeholder}'
    if (!reader) return
    let raw = ''
    while (true) {
        const res = await reader.read()
        if (res.done) return
        raw = `${raw}${new TextDecoder().decode(res.value)}`
        if (!raw.endsWith('}') && !raw.endsWith(']')) continue
        try {
            const text = raw.replaceAll('}{', `}${placeholder}{`).split(placeholder)
            yield JSON.parse(text[text.length - 1]!)
        } catch (_) {}
        raw = ''
    }
}
