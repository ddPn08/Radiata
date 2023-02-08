import { Box, Center, createStyles, Flex, MediaQuery, Navbar, Stack } from '@mantine/core'
import { Tab } from '../types/tab'
import GithubButton from './githubButton'
import ThemeToggle from './themeToggle'

interface Props {
  current?: string | number | undefined
  onChange?: (tab: string) => void
  tabs: Tab[]
}

const useStyles = createStyles((theme, _params, getRef) => {
  const icon = getRef('icon')
  return {
    link: {
      ...theme.fn.focusStyles(),
      display: 'flex',
      alignItems: 'center',
      textDecoration: 'none',
      fontSize: theme.fontSizes.sm,
      color: theme.colorScheme === 'dark' ? theme.colors.dark[1] : theme.colors.gray[7],
      padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
      borderRadius: theme.radius.sm,
      fontWeight: 500,
      cursor: 'pointer',
      gap: '4px',

      '&:hover': {
        backgroundColor: theme.colorScheme === 'dark' ? theme.colors.dark[6] : theme.colors.gray[0],
        color: theme.colorScheme === 'dark' ? theme.white : theme.black,

        [`& .${icon}`]: {
          color: theme.colorScheme === 'dark' ? theme.white : theme.black,
        },
      },
    },

    linkActive: {
      '&, &:hover': {
        backgroundColor: theme.fn.variant({ variant: 'light', color: theme.primaryColor })
          .background,
        color: theme.fn.variant({ variant: 'light', color: theme.primaryColor }).color,
        [`& .${icon}`]: {
          color: theme.fn.variant({ variant: 'light', color: theme.primaryColor }).color,
        },
      },
    },
  }
})

const Tabs = ({ current, onChange, tabs }: Props) => {
  const { classes, cx } = useStyles()

  const largeLinks = tabs.map((item) => (
    <Box
      className={cx(classes.link, { [classes.linkActive]: item.id === current })}
      key={item.id}
      onClick={(event) => {
        event.preventDefault()
        onChange && onChange(item.id)
      }}
    >
      <item.icon />
      <span>{item.label}</span>
    </Box>
  ))

  const smallLinks = tabs.map((item) => (
    <Center
      className={cx(classes.link, { [classes.linkActive]: item.id === current })}
      key={item.id}
      onClick={(event) => {
        event.preventDefault()
        onChange && onChange(item.id)
      }}
    >
      <item.icon />
    </Center>
  ))

  return (
    <>
      {/* large */}
      <MediaQuery
        smallerThan={'md'}
        styles={{
          display: 'none',
        }}
      >
        <Navbar height={'100%'} w={'240px'} p="md">
          <Navbar.Section grow>{largeLinks}</Navbar.Section>

          <Navbar.Section>
            <Flex justify={'space-between'}>
              <GithubButton />
              <ThemeToggle />
            </Flex>
          </Navbar.Section>
        </Navbar>
      </MediaQuery>

      {/* small */}
      <MediaQuery
        largerThan={'md'}
        styles={{
          display: 'none',
        }}
      >
        <Navbar height={'100%'} w={'100px'} p="md">
          <Navbar.Section grow>{smallLinks}</Navbar.Section>

          <Navbar.Section>
            <Stack>
              <GithubButton />
              <ThemeToggle />
            </Stack>
          </Navbar.Section>
        </Navbar>
      </MediaQuery>
    </>
  )
}

export default Tabs
