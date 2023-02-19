export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    if (!reader) return
    let raw = ''
    while (true) {
        const res = await reader.read()
        if (res.done) break
        try {
            raw += new TextDecoder().decode(res.value)
            const line = raw.split('\n')
            for (const text of line.slice(0, -1)) {
                yield JSON.parse(text)
            }
            raw = line[line.length - 1]!
        } catch (e) {
            console.error(e)
        }
    }
    if (raw.length > 0) {
        yield JSON.parse(raw)
    }
}
