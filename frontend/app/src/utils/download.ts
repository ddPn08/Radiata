export const downloadFile = async (url: string, filename?: string) => {
    const response = await fetch(url)
    const blob = await response.blob()
    const objectURL = URL.createObjectURL(blob)

    const link = document.createElement('a')
    link.href = objectURL
    link.download = filename || ''
    link.click()

    URL.revokeObjectURL(objectURL)
}

export const downloadB64 = async (b64: string, filename: string, contentType: string) => {
    const bin = atob(b64.replace(/^.*,/, ''))
    const bufArr = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) {
        bufArr[i] = bin.charCodeAt(i)
    }
    const file = new File([bufArr], filename)
    const url = URL.createObjectURL(file)
    const a = document.createElement('a')
    a.href = `data:${contentType};base64,${b64}`
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
}
