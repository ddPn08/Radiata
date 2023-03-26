import { MantineProvider, Flex } from '@mantine/core'
import { useAtomValue } from 'jotai'

import { themeAtom } from './atoms/theme'
import { Tabs } from './components/tabs'

export const App = () => {
  const theme = useAtomValue(themeAtom)

  return (
    <MantineProvider
      theme={{
        colorScheme: theme!,
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
        <Tabs />
      </Flex>
    </MantineProvider>
  )
}
