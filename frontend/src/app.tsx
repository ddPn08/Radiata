import { MantineProvider, Box, Flex } from '@mantine/core'
import { IconEngine, IconPhotoEdit, IconPhotoSearch } from '@tabler/icons-react'
import { useAtomValue } from 'jotai'
import { useState } from 'react'

import { themeAtom } from './atoms/theme'
import Tabs from './components/tabs'
import Engine from './tabs/engine'
import Generator from './tabs/generator'
import ImagesBrowser from './tabs/imagesBrowser'
import { Tab, TabType } from './types/tab'

const TABS: Tab[] = [
  {
    id: 'generator',
    label: 'Generator',
    icon: IconPhotoEdit,
  },
  {
    id: 'engine',
    label: 'Engine',
    icon: IconEngine,
  },
  {
    id: 'imagesBrowser',
    label: 'Images Browser',
    icon: IconPhotoSearch,
  },
]

const PAGES: Record<TabType, JSX.Element> = {
  generator: <Generator />,
  engine: <Engine />,
  imagesBrowser: <ImagesBrowser />,
}

const App = () => {
  const theme = useAtomValue(themeAtom)

  const [currentTab, setCurrentTab] = useState(TABS[0].id)

  return (
    <MantineProvider
      theme={{
        colorScheme: theme,
      }}
      withGlobalStyles
      withNormalizeCSS
    >
      <Flex
        h={'100svh'}
        sx={{
          overflowY: 'hidden',
        }}
      >
        <Tabs
          current={currentTab}
          tabs={TABS}
          onChange={(id) => {
            setCurrentTab(id as TabType)
          }}
        />
        {Object.keys(PAGES).map((key) => {
          return (
            <Box
              key={key}
              sx={{
                display: currentTab === key ? 'block' : 'none',
              }}
              w={'100%'}
            >
              {PAGES[key as TabType]}
            </Box>
          )
        })}
      </Flex>
    </MantineProvider>
  )
}

export default App
