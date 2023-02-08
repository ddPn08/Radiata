import { ActionIcon, Center } from '@mantine/core'
import { IconMoon, IconSun } from '@tabler/icons-react'
import { useAtom } from 'jotai'
import { themeAtom } from '../atoms/theme'

const ThemeToggle = () => {
  const [theme, setTheme] = useAtom(themeAtom)
  return (
    <Center>
      <ActionIcon
        onClick={() => {
          setTheme(theme === 'dark' ? 'light' : 'dark')
        }}
      >
        {theme === 'dark' ? <IconSun /> : <IconMoon />}
      </ActionIcon>
    </Center>
  )
}

export default ThemeToggle
