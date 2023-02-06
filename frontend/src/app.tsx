import { Box, Flex } from '@mantine/core'
import { Tab } from './types/tab'
import Tabs from './components/tabs'
import Txt2Img from './tabs/txt2img'
import { useEffect, useState } from 'react'
import Engine from './tabs/engine'
import { IconEngine, IconPhotoEdit } from '@tabler/icons-react'

const TABS: Tab[] = [
  {
    id: 'txt2img',
    label: 'Text to Image',
    icon: IconPhotoEdit,
  },
  {
    id: 'engine',
    label: 'Engine',
    icon: IconEngine,
  },
]

const PAGES: Record<string, () => JSX.Element> = {
  txt2img: Txt2Img,
  engine: Engine,
}

const App = () => {
  const [currentTab, setCurrentTab] = useState(TABS[0].id)
  const [currentTabComponent, setCurrentTabComponent] = useState(PAGES[currentTab])

  useEffect(() => {
    setCurrentTabComponent(PAGES[currentTab])
  }, [currentTab])

  return (
    <Flex h={'100vh'}>
      <Tabs
        current={currentTab}
        tabs={TABS}
        onChange={(id) => {
          setCurrentTab(id)
        }}
      />
      <Box w={'100%'}>{currentTabComponent && currentTabComponent}</Box>
    </Flex>
  )
}

export default App
