import { ActionIcon, Box, Image } from '@mantine/core'
import { IconDownload } from '@tabler/icons-react'

import { GeneratedImage } from '~/types/generatedImage'
import { downloadFile } from '~/utils/download'

interface Props {
  image: GeneratedImage
  onClick: () => void
}

const GalleryImage = ({ image, onClick }: Props) => {
  return (
    <Box key={image.url} pos={'relative'}>
      <Image
        src={image.url}
        alt={image.info.prompt}
        sx={{
          cursor: 'pointer',
        }}
        onClick={() => {
          onClick()
        }}
      />

      <ActionIcon
        variant={'light'}
        color={'gray'}
        pos={'absolute'}
        m={'xs'}
        right={0}
        bottom={0}
        onClick={() => {
          downloadFile(image.url)
        }}
      >
        <IconDownload size={16} />
      </ActionIcon>
    </Box>
  )
}

export default GalleryImage
