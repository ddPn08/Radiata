import Color from 'color'
import { createThemeStore, DecoRockProvider, type DefaultTheme } from 'decorock'
import { Component, createEffect, JSX } from 'solid-js'

import './global.css'

const MediaBreakpoints = {
  sm: '(min-width: 512px)',
  md: '(min-width: 768px)',
  lg: '(min-width: 1024px)',
  xl: '(min-width: 1420px)',
} as const

const themes: Record<'dark' | 'light', DefaultTheme> = {
  dark: {
    name: 'dark',
    colors: {
      primary: Color('#eee'),
      secondary: Color('#222'),
    },
    media: {
      breakpoints: MediaBreakpoints,
    },
  },
  light: {
    name: 'light',
    colors: {
      primary: Color('#222'),
      secondary: Color('#eee'),
    },
    media: {
      breakpoints: MediaBreakpoints,
    },
  },
}

export const [theme, setTheme] = createThemeStore({ ...themes['dark'] })

const GlobalStyles: Component = () => {
  return (
    <style
      // eslint-disable-next-line solid/no-innerhtml
      innerHTML={`
        body {
          background-color: ${theme.colors.secondary};
          color: ${theme.colors.primary};
          min-height: 100vh;
        }
      `}
    />
  )
}

export const ThemeProvider: Component<{ children: JSX.Element }> = (props) => {
  createEffect(() => {
    setTheme({ ...themes['dark'] })
  })

  return (
    <DecoRockProvider theme={theme} build={(p) => (p?.string ? p : p)}>
      <GlobalStyles />
      {props.children}
    </DecoRockProvider>
  )
}

declare module 'decorock' {
  export interface DefaultTheme {
    name: string
    colors: {
      primary: Color
      secondary: Color
    }
    media: {
      breakpoints: {
        sm: string
        md: string
        lg: string
        xl: string
      }
    }
  }
}
