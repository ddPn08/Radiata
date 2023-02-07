import { css } from 'decorock'
import { createSignal } from 'solid-js'

import { Header } from './components/header'
import { Tabs, TabPanel } from './components/ui/tabs'
import { ToastProvider } from './components/ui/toast'
import { ThemeProvider } from './styles'
import { Engine } from './tabs/engine'
import { Img2Img } from './tabs/img2img'
import { Txt2Img } from './tabs/txt2img'

export const App = () => {
  return (
    <ThemeProvider>
      <ToastProvider>
        <Index />
      </ToastProvider>
    </ThemeProvider>
  )
}

const PAGES = {
  txt2img: Txt2Img,
  img2img: Img2Img,
  engine: Engine,
}

const Index = () => {
  const [current, setCurrent] = createSignal('txt2img')

  return (
    <div
      class={css`
        display: grid;
        height: 100vh;
        grid-template-columns: 100%;
        grid-template-rows: 75px 1fr;
        overflow-y: auto;
      `}
    >
      <Header />
      <Tabs
        tab={current()}
        onChange={setCurrent}
        tabs={PAGES}
        vertical
        component={([label, Comp], isSelected) => {
          return (
            <TabPanel
              class={css`
                padding: 1rem;
                overflow-y: auto;
                transition: 0.2s;
              `}
              show={isSelected()}
              unmount={label !== 'txt2img'}
            >
              <Comp />
            </TabPanel>
          )
        }}
      />
    </div>
  )
}
