import { Box, Center, createStyles, Flex, MediaQuery, Navbar, Space, Stack } from '@mantine/core'
import { IconEngine, IconPhotoEdit, IconPhotoSearch } from '@tabler/icons-react'
import type { Property } from 'csstype'
import React, { useState } from 'react'

import { GithubButton } from './githubButton'
import { ThemeToggle } from './themeToggle'
import type { Tab } from '../types/tab'

import { plugins } from '~/plugin/pluginLoader'
import { Engine } from '~/tabs/engine'
import { Generator } from '~/tabs/generator'
import { ImagesBrowser } from '~/tabs/imagesBrowser'

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
          .background as Property.BackgroundColor,
        color: theme.fn.variant({ variant: 'light', color: theme.primaryColor })
          .color as Property.Color,
        [`& .${icon}`]: {
          color: theme.fn.variant({ variant: 'light', color: theme.primaryColor })
            .color as Property.Color,
        },
      },
    },
  }
})

const TABS: Tab[] = [
  {
    id: 'generator',
    label: 'Generator',
    icon: IconPhotoEdit,
    component: Generator,
  },
  {
    id: 'engine',
    label: 'Engine',
    icon: IconEngine,
    component: Engine,
  },
  {
    id: 'imagesBrowser',
    label: 'Images Browser',
    icon: IconPhotoSearch,
    component: ImagesBrowser,
  },
]

const LargeLink: React.FC<{
  item: Tab
  selected: boolean
  onClick: React.ComponentProps<'div'>['onClick']
}> = ({ item, selected, onClick }) => {
  const { classes, cx } = useStyles()
  return (
    <Box
      className={cx(classes['link'], { [classes['linkActive']!]: selected })}
      key={item.id}
      onClick={onClick}
    >
      <item.icon />
      <span>{item.label}</span>
    </Box>
  )
}
const SmallLink: React.FC<{
  item: Tab
  selected: boolean
  onClick: React.ComponentProps<'div'>['onClick']
}> = ({ item, selected, onClick }) => {
  const { classes, cx } = useStyles()
  return (
    <Center
      className={cx(classes['link'], { [classes['linkActive']!]: selected })}
      key={item.id}
      onClick={onClick}
    >
      <item.icon />
    </Center>
  )
}

export const Tabs = () => {
  const [current, setCurrent] = useState(TABS[0]!.id)

  const largeLinks = (
    <>
      {TABS.map((item) => (
        <LargeLink
          key={item.id}
          item={item}
          selected={item.id === current}
          onClick={(event) => {
            event.preventDefault()
            setCurrent(item.id)
          }}
        />
      ))}
      {plugins.map((plugin) => (
        <React.Fragment key={plugin.meta.name}>
          <Space h={'md'} />
          <Center>{plugin.meta.name}</Center>
          <Space h={'sm'} />
          {plugin.data.tabs.map((tab) => {
            const id = `${plugin.meta.name}-${tab.id}`
            return (
              <LargeLink
                key={id}
                item={tab as Tab}
                selected={id === current}
                onClick={(event) => {
                  event.preventDefault()
                  setCurrent(id)
                }}
              />
            )
          })}
        </React.Fragment>
      ))}
    </>
  )

  const smallLinks = (
    <>
      {TABS.map((item) => (
        <SmallLink
          key={item.id}
          item={item}
          selected={item.id === current}
          onClick={(event) => {
            event.preventDefault()
            setCurrent(item.id)
          }}
        />
      ))}
      {plugins.map((plugin) => (
        <React.Fragment key={plugin.meta.name}>
          {plugin.data.tabs.map((tab) => {
            const id = `${plugin.meta.name}-${tab.id}`
            return (
              <SmallLink
                key={id}
                item={tab as Tab}
                selected={id === current}
                onClick={(event) => {
                  event.preventDefault()
                  setCurrent(id)
                }}
              />
            )
          })}
        </React.Fragment>
      ))}
    </>
  )

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

      {TABS.map((v) => (
        <Box
          key={v.id}
          sx={{
            display: current === v.id ? 'block' : 'none',
          }}
          w={'100%'}
        >
          {React.createElement(v.component)}
        </Box>
      ))}
      {plugins.map((plugin) => (
        <React.Fragment key={plugin.meta.name}>
          {plugin.data.tabs.map((tab) => {
            const id = `${plugin.meta.name}-${tab.id}`
            return (
              <Box
                key={id}
                sx={{
                  display: current === id ? 'block' : 'none',
                }}
                w={'100%'}
              >
                {React.createElement(tab.component)}
              </Box>
            )
          })}
        </React.Fragment>
      ))}
    </>
  )
}
