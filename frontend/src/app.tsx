import { Box, Flex } from '@mantine/core'
import { Tab } from './types/tab'
import Tabs from './components/tabs'
import Txt2Img from './tabs/txt2img'
import { useState } from 'react'
import Engine from './tabs/engine'
import { IconEngine, IconPhotoEdit } from '@tabler/icons-react'

const TABS: Tab[] = [
  {
    id: 'txt2img',
    label: 'Text to Image',
    icon: <IconPhotoEdit />,
    component: Txt2Img,
  },
  {
    id: 'engine',
    label: 'Engine',
    icon: <IconEngine />,
    component: Engine,
  },
]

const App = () => {
  const [currentTab, setCurrentTab] = useState(TABS[0].id)
  const [currentTabComponent, setCurrentTabComponent] = useState(TABS[0].component)

  return (
    <Flex h={'100vh'}>
      <Tabs
        tab={currentTab}
        tabs={TABS}
        onChange={([id, index]) => {
          setCurrentTab(id)
          setCurrentTabComponent(TABS[index].component)
        }}
      />
      <Box w={'100%'}>{currentTabComponent && currentTabComponent}</Box>
    </Flex>
  )
}

export default App
