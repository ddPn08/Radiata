import Color from 'color'
import { styled } from 'decorock'
import { Component, ComponentProps, splitProps } from 'solid-js'

export const Container = styled.label`
  display: inline-block;
  padding: 0.25rem 0.5rem;
  border-radius: 0.5rem;
  margin-bottom: 0.5rem;
  background-color: ${() => Color('red').lighten(0.25)};
  color: white;
  font-size: 0.75rem;
  font-weight: 400;
`
export const Required: Component<ComponentProps<'label'>> = (props) => {
  const [local, others] = splitProps(props, ['children'])
  return <Container {...others}>{local.children || '必須'}</Container>
}
