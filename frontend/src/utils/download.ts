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
