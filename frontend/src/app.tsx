import { MantineProvider, Box, Flex } from '@mantine/core'
import { Tab } from './types/tab'
import Tabs from './components/tabs'
import Generator from './tabs/gemerator'
import { useState } from 'react'
import Engine from './tabs/engine'
import { IconEngine, IconPhotoEdit } from '@tabler/icons-react'
import { useAtomValue } from 'jotai'
import { themeAtom } from './atoms/theme'

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
]

const PAGES: Record<string, JSX.Element> = {
  generator: <Generator />,
  engine: <Engine />,
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
      <Flex h={'100vh'}>
        <Tabs
          current={currentTab}
          tabs={TABS}
          onChange={(id) => {
            setCurrentTab(id)
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
              {PAGES[key]}
            </Box>
          )
        })}
      </Flex>
    </MantineProvider>
  )
}

export default App
