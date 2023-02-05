import autosize from 'autosize'
import { styled } from 'decorock'
import { Component, ComponentProps, createSignal, onMount, splitProps } from 'solid-js'

const StyledTextarea = styled.textarea`
  display: inline-block;
  width: 100%;
  box-sizing: border-box;
  padding: 0.5rem;
  border: 1px solid ${(p) => p.theme.colors.primary.fade(0.5).string()};
  border-radius: 0.5rem;
  background-color: rgba(0, 0, 0, 0.05);
  color: ${(p) => (p.disabled ? p.theme.colors.primary.fade(0.25) : p.theme.colors.primary)};
  font-size: 0.8rem;
  outline: none;

  &:focus {
    border: 1px solid ${(p) => p.theme.colors.primary.darken(0.25).string()};
  }
`

export const Textarea: Component<ComponentProps<'textarea'>> = (props) => {
  const [ref, setRef] = createSignal<HTMLTextAreaElement>()
  const [local, others] = splitProps(props, ['children'])

  onMount(() => {
    const el = ref()!
    if (props.ref) {
      if (typeof props.ref === 'function') props.ref(el)
      else props.ref = el
    }
    autosize(el)
  })

  return (
    <StyledTextarea ref={setRef} {...others}>
      {local.children}
    </StyledTextarea>
  )
}
