import { ActionIcon } from '@mantine/core'
import { IconBrandGithub } from '@tabler/icons-react'

const GithubButton = () => {
  return (
    <a href={'https://github.com/ddPn08/Lsmith'} target={'_blank'} rel="noreferrer">
      <ActionIcon>
        <IconBrandGithub />
      </ActionIcon>
    </a>
  )
}

export default GithubButton
