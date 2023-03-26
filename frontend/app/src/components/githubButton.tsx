import { ActionIcon, Center } from '@mantine/core'
import { IconBrandGithub } from '@tabler/icons-react'

export const GithubButton = () => {
  return (
    <Center>
      <a href={'https://github.com/ddPn08/Lsmith'} target={'_blank'} rel="noreferrer">
        <ActionIcon>
          <IconBrandGithub />
        </ActionIcon>
      </a>
    </Center>
  )
}
