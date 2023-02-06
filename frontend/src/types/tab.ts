export interface Tab {
  id: string
  label: string
  icon: JSX.Element
  component: () => JSX.Element
}
