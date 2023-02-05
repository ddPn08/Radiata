import { css, styled } from 'decorock'
import { createSignal, onMount } from 'solid-js'

import { IconButton } from './ui/icon-button'
import { Select } from './ui/select'

import { api } from '~/api'
import { events } from '~/events'
import IconRefresh from '~icons/material-symbols/refresh'

const Container = styled.div`
  height: 100%;
  padding: 1rem;
  box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
`

export const Header = () => {
  const [current, setCurrent] = createSignal<string>('')
  const [models, setModels] = createSignal<string[]>([])
  const [selecting, setSelecting] = createSignal(false)

  const reload = () => {
    api.getRunners().then((res) => {
      setModels(res.data)
    })
    api.getCurrentRunner().then((res) => {
      setCurrent(res.data)
    })
  }

  onMount(() => {
    reload()
  })

  return (
    <Container>
      <div
        class={css`
          display: flex;
          width: 40%;
          align-items: center;
        `}
      >
        <div
          class={css`
            width: 100%;
          `}
        >
          <Select
            options={models().map((v) => ({ label: v, value: v }))}
            value={current()}
            onChange={async (item) => {
              if (selecting()) return
              setSelecting(true)
              setCurrent(item.value)
              await api.setRunner({ setRunnerRequest: { model_id: item.value } })
              events.emit('model-select', item.value)
              setSelecting(false)
            }}
          />
        </div>
        <IconButton onClick={reload}>
          <IconRefresh />
        </IconButton>
      </div>
    </Container>
  )
}
