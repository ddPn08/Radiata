export const streamGenerator = async function* (body: ReadableStream<Uint8Array>) {
    const reader = body?.getReader()
    if (!reader) return
    let raw = ''
    while (true) {
        const res = await reader.read()
        if (res.done) return
        try {
            raw += new TextDecoder().decode(res.value)
            if (raw.endsWith('\n')) {
                const line = raw.split('\n')
                yield JSON.parse(line[line.length - 2]!)
                raw = ''
            }
        } catch (e) {
            console.error(e)
        }
    }
}
