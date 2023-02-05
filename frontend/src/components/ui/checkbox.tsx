import { styled } from 'decorock'
import { Component, ComponentProps, splitProps } from 'solid-js'

const Container = styled.div`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  gap: 1rem;

  p {
    font-size: 1rem;
    font-weight: bold;
    user-select: none;
  }

  input {
    cursor: pointer;
    transform: scale(1.5);
  }
`

const Input: Component<Omit<ComponentProps<'input'>, 'type'>> = (props) => {
  const [local, others] = splitProps(props, ['children', 'class', 'ref'])
  // eslint-disable-next-line prefer-const
  let ref = local.ref as HTMLInputElement
  return (
    <Container
      class={local.class}
      onClick={(e) => {
        if (ref !== e.target) ref.click()
      }}
    >
      <input ref={ref!} type="checkbox" {...others} />
      <p>{local.children}</p>
    </Container>
  )
}

export const CheckBox = styled(Input)``
