import { createPlugin } from '@lsmith/api'

export default createPlugin(() => {
    return {
        tabs: [
            {
                id: "hw",
                label: "Hello World",
                icon: () => <>☆</>,
                component: () => <h1>Hello World</h1>
            }
        ]
    }
})